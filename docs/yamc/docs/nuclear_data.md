# Nuclear data



### Config the nuclear data

This minimal example uses a single nuclear data library for every nuclide's
neutron data and every element's photon data. Other configurations are covered
under the nuclear data options below.

```python
yamc.neutron_cross_section_data = "endf-b8.1"
yamc.photon_cross_section_data = "endf-b8.1"
```

A run that transports only neutrons needs only the first line, and one that
transports only photons needs only the second.

### Nuclear data settings

Six module-level settings, read and written like attributes. Each takes a library
keyword or a path, and each can point somewhere different, so a run can mix
libraries by particle and by subsection.

| setting | supplies | decides | left unset |
| --- | --- | --- | --- |
| ``neutron_cross_section_data`` | neutron cross sections per nuclide, and the thermal scattering tables that go with them | reaction rates in neutron transport and transmutation | no default library; set it or provide explicit material data |
| ``photon_cross_section_data`` | photoatomic photon data per element | photon interactions in photon transport | no default library; set it or provide explicit material data |
| ``transmutation_reactions`` | which reaction on which nuclide gives which product, and its Q | which activation products exist to be made | no default: a calculation that reads it is refused |
| ``transmutation_decay_data`` | half-lives, decay modes, mean decay energies, decay photon lines | how the inventory decays, and the photon source a shutdown dose rate transports | no default: a calculation that reads it is refused |
| ``transmutation_fission_yields`` | fission product yields | what a fission makes | no default: a calculation that reads it is refused |
| ``transmutation_branch_ratios`` | isomeric branching against incident energy | which product is left in a metastable state | no overlay, so every product lands in its ground state |

Only ``transmutation_branch_ratios`` has a default. The other five name the
data a calculation reads, so nothing is downloaded from a library you did not
choose: a calculation that needs a setting you have left unset is refused before
it starts, with a message naming each unset setting and the line to copy. Every
transmutation and shutdown-dose example on these pages sets what it reads.
Which transmutation settings a calculation reads is listed under
[Transmutation data](#transmutation-data).

``neutron_cross_section_data`` and ``transmutation_reactions`` are the pair most
often confused. They are not alternatives: the cross sections decide how fast each
reaction runs, and the reactions network, which holds no cross sections, decides
which reactions and products exist. See
[The two that both say reactions](#the-two-that-both-say-reactions).

#### Neutron and photon cross sections

``neutron_cross_section_data`` and ``photon_cross_section_data`` are
independent. Both start unset, and setting one never sets, reads or changes the
other, so each particle's data comes from the library you name for it. Each
takes a library keyword, a local directory, or a dict: keyed by nuclide
(``"Fe56"``) for neutrons and by element (``"Fe"``) for photons. Assigning a
value replaces that particle's whole selection, and ``None`` or an empty dict
clears it. What a calculation reads:

| calculation | cross-section settings it reads |
| --- | --- |
| neutron transport | ``neutron_cross_section_data`` |
| transport from a ``PhotonSource`` | ``photon_cross_section_data`` |
| coupled neutron-photon transport (``transport_secondary_photons=True``) | both |
| a D1S shutdown dose run (``use_decay_photons=True``) | both, plus the decay data and reactions network |
| an irradiation: ``Material.transmute()``, ``yamc.transmute()`` and ``Model.simulate_transmutation()`` | ``neutron_cross_section_data``, plus the transmutation settings |
| activity, decay heat, decay photon lines and contact dose, and a schedule that only cools | neither, only ``transmutation_decay_data`` |

In a coupled run the photons a neutron makes come from the neutron evaluation,
and ``photon_cross_section_data`` decides how those photons then interact. The
data are loaded automatically when a calculation needs them, and a calculation
missing the data it needs is refused before it starts, naming the setting and
the nuclides or elements without a source; no other library is substituted.
Explicit per-material sources are available too. See
[Nuclear data libraries](nuclear_data_libraries.md#photon-data-is-selected-separately)
for which libraries publish photon data.

#### The two that both say reactions

``neutron_cross_section_data`` and ``transmutation_reactions`` are halves of one
calculation rather than alternatives. For each nuclide the network lists the
channels it has, each channel's name is mapped to an MT number, and the cross
sections are what that MT is priced with. So the network decides which channels
are considered at all and the cross sections decide how fast each one runs.
Swap the library behind ``transmutation_reactions`` and products appear or
disappear; swap the one behind ``neutron_cross_section_data`` and the same products
arrive at different rates.

They also fail differently, which is how to tell them apart in practice. A missing
``neutron_cross_section_data`` stops a neutron transport, an irradiation or a
cross-section query before it starts, with an error naming the setting and a
nuclide it has no data for; a decay-only schedule needs no cross sections and still runs. A
network pointed at the wrong material is refused before the solve, naming what
the material holds against what the network drives.

### Nuclide properties and cross sections

Nuclides can be created and their properties inspected.

```python
nuclide = yamc.Nuclide('Li6')

print(nuclide.name)                    # "Li6"
print(nuclide.element)                 # "lithium"
print(nuclide.atomic_number)           # 3
print(nuclide.mass_number)             # 6
print(nuclide.available_temperatures)  # ["250", "294", "600", "900", "1200", "2500"]
print(nuclide.reaction_mts)            # [1, 2, 3, 4, 27, 32, ..., 444, 901]
```

Microscopic cross sections are retrieved by reaction name or MT number. Each call returns ``(xs_values, energy_grid)``. Different reactions may have different energy grids.

```python
xs_total, energy = nuclide.microscopic_cross_section(reaction=1, temperature="294")           # total
xs_elastic, energy = nuclide.microscopic_cross_section(reaction=2, temperature="294")         # elastic
xs_ngamma, energy = nuclide.microscopic_cross_section(reaction="(n,gamma)", temperature="294")
xs_nt, energy = nuclide.microscopic_cross_section(reaction="(n,t)", temperature="294")
```

Plotting cross sections with matplotlib (the figure below is rendered live from this code when the docs are built):

```python exec="1" html="1" source="above"
import matplotlib.pyplot as plt
import yamc

yamc.neutron_cross_section_data = "endf-b8.1"

nuclide = yamc.Nuclide("Li6")
reactions = [
    (1, "Total (MT 1)"),
    (2, "Elastic (MT 2)"),
    ("(n,gamma)", "(n,gamma)"),
    ("(n,t)", "(n,t)"),
]

fig, ax = plt.subplots(figsize=(8, 5))
for reaction, label in reactions:
    xs, energy = nuclide.microscopic_cross_section(reaction=reaction, temperature="294")
    ax.loglog(energy, xs, label=label)

ax.set_xlabel("Energy [eV]")
ax.set_ylabel("Microscopic Cross Section [barns]")
ax.set_title("Li6 microscopic cross sections")
ax.legend()
ax.grid(True, alpha=0.3)

from io import StringIO as _S
_b = _S(); plt.savefig(_b, format="svg", bbox_inches="tight"); print(_b.getvalue())
```

### Elements and natural abundances

Element data is available via the ``yamc.data`` module.

```python
# Natural abundances
abundances = yamc.data.natural_abundance()
print(abundances["Li6"])  # 0.07589
print(abundances["Li7"])  # 0.92411

# List nuclides for an element
nuclides = yamc.data.element_nuclides()
print(nuclides["Li"])  # ["Li6", "Li7"]
```

The abundances are the IUPAC TICE 2013 isotopic compositions (Meija et al.,
Pure Appl. Chem. 88 (2016) 293, doi:10.1515/pac-2015-0503). Each value is the
column 9 representative isotopic composition. The 12 elements that TICE gives
as an interval rather than a value (H, Li, B, C, N, O, Mg, Si, S, Cl, Br, Tl)
use the column 6 best measurement instead, and mononuclidic elements are 1.

``natural_abundance_records()`` returns the full TICE row behind each value:
the column 9 value and its expanded uncertainty (or the interval), the column 4
observed range of natural variation, and the column 6 best measurement with its
uncertainty, coverage factor and calibration flag. A field TICE leaves empty is
``None``, which means "not stated", never zero.

```python
records = yamc.data.natural_abundance_records()
fe58 = records["Fe58"]
print(fe58["representative_value"], fe58["representative_uncertainty"])  # 0.00282 0.00012
print(fe58["best_measurement"], fe58["best_measurement_uncertainty"])    # 0.002819 2.7e-05
print(fe58["best_measurement_coverage"])                                 # '2s'
print(records["Li6"]["representative_interval"])                         # (0.019, 0.078)
```

These uncertainties are reference data only. No calculation samples or
propagates them: turning them into a distribution needs choices TICE does not
make (a coverage factor for column 9, correlations between the isotopes of an
element, how to treat an interval).

The element symbol table is bound too, so a script naming files or directories
from ENDF data does not need its own copy of the periodic table.

```python
yamc.data.atomic_symbol(26)         # 'Fe'
yamc.data.atomic_number("Fe")       # 26
yamc.data.split_nuclide("Fe56")     # ('Fe', 56, 0)
yamc.data.split_nuclide("Ta180_m1") # ('Ta', 180, 1)
```

``split_nuclide`` returns ``(symbol, mass number, isomeric state)``, the state
being 0 for a ground state and 1 for the first metastable. Splitting a name by
stripping its trailing digits does not do this: that leaves ``"Ta180_m"`` with a
mass number of 1, and natural tantalum really does contain Ta180m, so a
composition built from ``natural_abundance()`` will hand you that name.

``atomic_symbol`` takes Z = 0 and answers ``'n'`` for the neutron rather than
raising, because ENDF libraries really do ship a neutron material (JENDL is one)
and a caller walking a sublibrary will meet it. ``atomic_number`` is case
sensitive and takes the symbol alone: ``"fe"`` and ``"iron"`` are both refused.

Element and reaction labels are available for building tables and selectors:

```python
print(yamc.data.element_names()["Fe"])
print(yamc.data.reaction_names()[102])
print(yamc.cache_dir())
```

### Photon attenuation and converted-data inspection

The NIST photon response tables used by contact-dose calculations are also
public. They are bundled reference data, independent of the selected transport
library:

```python
iron_attenuation = yamc.data.mass_attenuation_coefficient(element="Fe")
air_absorption = yamc.data.mass_energy_absorption_coefficient(material="air")
print(iron_attenuation.interpolate(energy=1e6))  # cm2/g
print(air_absorption.interpolate(energy=1e6))   # cm2/g
```

After a conversion, inspect the individual nuclide or element directory without
constructing a transport model:

```python
nuclide_info = yamc.read_nuclide_from_arrow("converted/Fe56.arrow")
element_info = yamc.read_element_from_arrow("converted/Fe.arrow")
print(nuclide_info, element_info)
```

`yamc.data.clear_nuclide_cache()` clears the in-memory nuclide cache, not the
downloaded files. A material holding data can release its own references with
`material.release_nuclear_data()` before the next part of a large sweep.

### Options for setting nuclear data source

The source of nuclear data can be set at the nuclide, material and program level in a few ways.

#### Cross sections from a library keyword

The simplest method is to name a library for each particle. The two need not be
the same library: here neutrons use FENDL-3.2d and photons ENDF/B-VIII.1.

```python
yamc.neutron_cross_section_data = "fendl-3.2d"
yamc.photon_cross_section_data = "endf-b8.1"
```

There are six recognized keywords: [``endf-b8.1``][cite-endf-b8.1], [``jeff-4.0``][cite-jeff-4.0], [``jendl-5.0``][cite-jendl-5.0],
[``tendl-2025``][cite-tendl-2025], [``tendl-2017``][cite-tendl-2017] and [``fendl-3.2d``][cite-fendl-3.2d].
[Nuclear data libraries](nuclear_data_libraries.md) has what each one publishes,
how big each one is, which can serve as a photon source, and
[what to cite](nuclear_data_libraries.md#cite-the-nuclear-data-you-use) when you
use it: each library's evaluators ask to be cited wherever their data is used.

TENDL has no decay or fission-yield evaluations. Take ``decay``/``fission_yields``
from a library that has them while sourcing ``reactions`` and/or ``branching``
from TENDL. TENDL publishes no photoatomic data, so for coupled neutron/photon
transport set ``photon_cross_section_data`` to a library that does. The next
TENDL release adds
[photonuclear evaluated data](nuclear_data_libraries.md#photonuclear-data-in-the-next-tendl-release),
which does not replace photoatomic transport data. Pointing a transmutation part
at a library that does not provide it still fails immediately.

TENDL-2025's Li6 and Li7 make no tritium below 30 MeV, whether used as cross sections or as a ``reactions`` network, so use another library wherever lithium matters: see [Lithium in TENDL-2025](nuclear_data_libraries.md#lithium-in-tendl-2025).

A keyword resolves, once per process, to the library's current published
release: a dated folder such as ``2026-10-10.1`` with a manifest that lists every
file in it with its size and sha256. Whenever nuclear data is needed, yamc looks
for the file in that release's folder of its cache. If the file is there it is
used; otherwise it is downloaded first, checked against the manifest (a byte
range against its own hash), and only then kept. A file that does not match is
refused with an error naming it. A load of many nuclides downloads them in
concurrent batches and prints a ``Downloaded k of N nuclides`` counter.

The cache is a ``yamc`` folder in the platform's per-user cache directory, and
``yamc.cache_dir()`` returns it:

| platform | cache directory |
| --- | --- |
| Linux | ``~/.cache/yamc``, or ``$XDG_CACHE_HOME/yamc`` when that is set |
| macOS | ``~/Library/Caches/yamc`` |
| Windows | ``%LOCALAPPDATA%\yamc`` |

Each release has its own folder, laid out as the published tree:

```
<cache dir>/<keyword>/<release>/manifest.json
<cache dir>/<keyword>/<release>/neutron/<Nuclide>.arrow/...
<cache dir>/<keyword>/<release>/photon/<Element>.arrow/...
<cache dir>/<keyword>/<release>/transmutation/<subsection>.arrow/...
```

No setting or environment variable moves the cache (``YAMC_CACHE_DIR`` is gone).
To read data from somewhere else, point the setting at a local directory, as
below, which bypasses the cache. When a newer release is published, the next run
downloads what it needs from it, says so once, and leaves the older folder in
place; every file one run reads from a library comes from a single release.
Without a connection, yamc uses the newest cached release that already holds
every file the run needs and says which; only a file missing from the cache is
then an error. Results record the release of each library they used:
``TransportResults.runs[i]["data_releases"]`` and
``TransmutationResults.data_releases``, each mapping a keyword to its
``release``, ``manifest_sha256``, ``format_version``, ``offline`` and
``converters`` (each build that wrote the release's files: the package, its
version and the core commit it was compiled from).
``yamc.combine_results`` refuses runs that read different releases of one
library.

#### Cross sections from a directory path

If all your Arrow data directories live in one directory you can point a setting at that directory.
When a nuclide is needed (e.g. `Li6`), YAMC resolves it to `<dir>/Li6.arrow` in the neutron directory, and an element (e.g. `Fe`) to `<dir>/Fe.arrow` in the photon directory.

<!-- doctest: skip -->
```python
yamc.neutron_cross_section_data = "/path/to/neutron/"
yamc.photon_cross_section_data = "/path/to/photon/"
```

#### A whole library in one folder

A folder laid out as a published release, with its ``manifest.json`` beside the
``neutron/``, ``photon/`` and ``transmutation/`` folders it describes, is a
release folder, and its one path serves every setting. Neutron cross sections
are read from ``neutron/``, photon data from ``photon/``, and each transmutation
part from ``transmutation/<part>.arrow/``:

<!-- doctest: skip -->
```python
yamc.neutron_cross_section_data = "/data/endf-b8.1"
yamc.photon_cross_section_data = "/data/endf-b8.1"
yamc.transmutation_decay_data = "/data/endf-b8.1"
yamc.transmutation_reactions = "/data/endf-b8.1"
yamc.transmutation_fission_yields = "/data/endf-b8.1"
```

A copy of every file a published release's ``manifest.json`` lists, from
``https://yamc-data.xsplot.com/<keyword>/<release>/``, with the manifest itself
at the top, is such a folder, and so is a release folder in the cache
(``<cache dir>/<keyword>/<release>/``) for the files it holds.

Nothing is downloaded. The manifest is read as a published release's is: a
``format_version`` this build does not read is refused with the advice to
upgrade, and each file a load reads is checked against the size the manifest
gives it. A part the manifest does not list is refused, naming the parts the
folder does carry, and so is a file the manifest lists that the folder does not
hold. Results record the folder in ``data_releases`` under its absolute path,
with its ``release`` and ``manifest_sha256``, and ``offline`` set to ``False``.

A load checks sizes only. ``yamc.verify_library`` is the full check: it
re-hashes every file the manifest lists and looks for files it does not list,
which is worth running on a library kept for offline use or copied between
machines:

<!-- doctest: skip -->
```python
import pprint

report = yamc.verify_library("/data/irdff-ii")
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

``ok`` is ``True`` when ``missing``, ``changed`` and ``unlisted`` are all empty.
An unlisted file fails the check, because results identify the library by the
manifest's sha256 and a file the manifest does not list is data that identity
does not cover. A release folder in the cache holds only the files runs have
downloaded, so the rest are reported missing.

A bare name that is both a keyword and a folder in the working directory
(``endf-b8.1`` with a folder ``./endf-b8.1`` beside it) is refused rather than
guessed at. Write ``./endf-b8.1`` or an absolute path to mean the folder.

#### Building a library yourself

``yamc.generate_library`` builds one of the published libraries from its
evaluated source archives into a folder laid out exactly as the published
release, with its ``manifest.json`` written last, so the folder serves every
setting as above:

<!-- doctest: skip -->
```python
report = yamc.generate_library("endf-b8.1", "/data/endf-b8.1")
print(report["release"], report["file_count"], report["njoy"])

yamc.neutron_cross_section_data = "/data/endf-b8.1"
yamc.photon_cross_section_data = "/data/endf-b8.1"
yamc.transmutation_decay_data = "/data/endf-b8.1"
yamc.transmutation_reactions = "/data/endf-b8.1"
yamc.transmutation_fission_yields = "/data/endf-b8.1"
yamc.transmutation_branch_ratios = "/data/endf-b8.1"
```

It downloads each source archive and checks it against the size and sha256 it
is pinned to, unpacks it, converts every evaluation (neutron data through NJOY
with every covariance section the evaluation has, photon data with the Compton
profile, density effect and bremsstrahlung tabulations, photonuclear data as
evaluated), builds the transmutation chain, reads every folder back through the
reader a simulation uses, and writes each particle's ``index.json`` and then the
manifest.

The libraries are [``"endf-b8.1"``][cite-endf-b8.1], [``"jeff-4.0"``][cite-jeff-4.0], [``"jendl-5.0"``][cite-jendl-5.0],
[``"tendl-2025"``][cite-tendl-2025], [``"tendl-2017"``][cite-tendl-2017], [``"fendl-3.2d"``][cite-fendl-3.2d] and [``"irdff-ii"``][cite-irdff-ii]. Each is
built from its own evaluations only, never with another library's data:

| keyword | particles | transmutation parts |
| --- | --- | --- |
| [``endf-b8.1``][cite-endf-b8.1], [``jendl-5.0``][cite-jendl-5.0] | neutron, photon | decay, reactions, fission_yields, branching |
| [``jeff-4.0``][cite-jeff-4.0] | neutron | decay, reactions, fission_yields, branching |
| [``tendl-2025``][cite-tendl-2025], [``tendl-2017``][cite-tendl-2017] | neutron, photonuclear | reactions, branching |
| [``fendl-3.2d``][cite-fendl-3.2d] | neutron, photon | reactions, branching |
| [``irdff-ii``][cite-irdff-ii] | none | branching |

A library without decay data of its own (TENDL, FENDL, IRDFF-II) writes each
excited product level of its branching as the evaluation gives it, and a run
names the isomer against the decay data it is given: TENDL branching with
``transmutation_decay_data = "endf-b8.1"`` is matched to ENDF/B-VIII.1's
isomers.

The keyword arguments narrow a build:

<!-- doctest: skip -->
```python
# The transmutation chain alone: no NJOY, minutes.
yamc.generate_library("endf-b8.1", "/data/endf-b8.1-chain",
                      parts=["decay", "reactions", "fission_yields", "branching"])

# A few nuclides and an element (every Fe evaluation, and Fe photon data).
yamc.generate_library("endf-b8.1", "/data/endf-b8.1-few",
                      nuclides=["Li6", "Li7", "Fe"], temperatures=[293.6, 600.0])
```

- ``parts``: of ``"neutron"``, ``"photon"``, ``"photonuclear"``, ``"decay"``,
  ``"reactions"``, ``"fission_yields"`` and ``"branching"``; the default is
  every part the library's release carries.
- ``nuclides``: nuclides and element symbols. The chain is always built whole.
- ``temperatures``: in Kelvin, by default 250, 293.6, 600, 900, 1200 and 2500 K.
  The 0 K grid is always kept.
- ``njoy_exec``: the NJOY to run. The default is the ``njoy`` the njoy2016
  wheel (``pip install njoy2016 --index-url https://shimwell.github.io/wheels``)
  installs beside this Python, then the one on PATH. FENDL needs the IAEA-NDS
  fork, and the builder refuses any other for it.
- ``downloads``: a folder of source archives fetched by hand, searched first.

Running it again resumes: a folder finished for this release is kept and only
the rest are converted. The release stamp is the date the build started, kept on
every resume. No manifest is written unless every evaluation converted (or is
one NJOY is known not to process, declared with the reason), every folder reads
back and the chain loads; a failed evaluation's reason is kept in
``<path>/.build/failed/``, and running again retries only those. The returned
dict has ``keyword``, ``release``, ``path``, ``parts``, ``converted`` and
``reused`` (``"<particle>/<Name>"`` keys), ``not_converted`` (the declared
failures with their reasons), ``stale_declarations``, ``njoy``,
``chain_nuclides``, ``manifest_sha256``, ``file_count`` and ``total_bytes``.

The release records the build that wrote it: ``manifest.json`` carries the
package and version (``provenance.builder``), the core commit
(``provenance.builder_commit``, as ``yamc.__commit__``) and whether the tree had
uncommitted changes (``provenance.builder_dirty``, as ``yamc.__commit_dirty__``),
and lists each distinct build as ``converters``. A resume must be the same
build, commit and dirty flag included, as the one that started the folder. An
extension compiled from uncommitted changes builds only a partial library (one
given ``parts``, ``nuclides`` or ``temperatures``), since no commit describes
what it would write; commit the changes and rebuild to build a whole one.

A full build is long, most of it NJOY: on a 32-core machine about 50 minutes
for ENDF/B-VIII.1, 9 hours for JENDL-5 and a day or more for each TENDL.

#### Converting single evaluations

The converters behind the builder take one evaluation at a time, for data of
your own or a patched evaluation. Each writes the same folders a library holds:

<!-- doctest: skip -->
```python
# Transport data, through NJOY, with the evaluation's covariance sections.
yamc.convert_neutron_transport(
    input_path="n-026_Fe_056.endf", output_dir="out/neutron",
    njoy_exec="njoy", temperatures=[293.6, 600.0], covariance=True)

# Cross sections only, for a rate collapse; from ENDF, or from ACE without NJOY.
yamc.convert_neutron_xs(
    input_path="n-026_Fe_056.endf", output_dir="out/neutron-xs")

# Photon data, with the tabulations yamc carries.
yamc.convert_photon("photoat-026_Fe_000.endf", "out/photon",
                    relaxation_path="atom-026_Fe_000.endf")

# A photonuclear evaluation (TENDL's gamma sublibrary), kept as evaluated.
yamc.convert_photonuclear(input_path="g-Fe056.tendl", output_dir="out/photonuclear")

# The transmutation chain, from decay, fission yield and neutron evaluations,
# and its isomeric branching.
yamc.convert_transmutation(
    decay_files=["dec-027_Co_060.endf"], fpy_files=["nfy-092_U_235.endf"],
    neutron_files=["n-027_Co_059.endf"], output_path="out/chain")
yamc.convert_branching(
    neutron_files=["n-027_Co_059.endf"], decay_files=["dec-027_Co_060.endf"],
    output_path="out/chain")
```

A folder of ``out/neutron/<Name>.arrow/`` directories is a path
``neutron_cross_section_data`` takes, a folder of ``out/photon/<Element>.arrow/``
directories one ``photon_cross_section_data`` takes, and ``out/chain`` a path
each transmutation setting takes. ``convert_transmutation`` and ``convert_branching`` read every
material in each file, so a tape of many materials needs no splitting, and they
name every product as the evaluation gives it: a product the decay data has no
record for is counted at run time as produced with unknown fate, never booked
onto a neighbouring nuclide. ``convert_branching`` given no decay files writes
each excited level as evaluated, to be matched at run time.

#### Cross sections from file paths

A nuclide's data is a folder of Arrow files, ``<Nuclide>.arrow``, which a published release serves at ``https://yamc-data.xsplot.com/<keyword>/<release>/neutron/<Nuclide>.arrow/`` (every file its ``manifest.json`` lists for that folder). Once such folders are saved locally, their paths can be used to configure the nuclear data. Replace the `tests` dir with the path to the downloaded data files.

```python
yamc.neutron_cross_section_data = {
    "Be9": "tests/Be9.arrow",
    "Fe54": "tests/Fe54.arrow",
    "Fe56": "tests/Fe56.arrow",
    "Fe57": "tests/Fe57.arrow",
    "Fe58": "tests/Fe58.arrow",
    "Li6": "tests/Li6.arrow",
    "Li7": "tests/Li7.arrow",
}
```

Photon data is keyed by element, and a published release serves each element at ``https://yamc-data.xsplot.com/<keyword>/<release>/photon/<Element>.arrow/``:

```python
yamc.photon_cross_section_data = {
    "Be": "tests/Be.arrow",
    "Fe": "tests/Fe.arrow",
    "Li": "tests/Li.arrow",
}
```

#### Cross sections from paths and specific libraries

Different sources can also be mixed. In this example we use the `endf-b8.1` and `fendl-3.2d` library keywords for some nuclides and local file paths for others. Replace the ```tests``` dir with the path to the downloaded data files.

```python
yamc.neutron_cross_section_data = {
    "Be9": "fendl-3.2d",
    "Fe54": "endf-b8.1",
    "Fe56": "tests/Fe56.arrow",
    "Fe57": "tests/Fe57.arrow",
    "Fe58": "tests/Fe58.arrow",
    "Li6": "endf-b8.1",
    "Li7": "fendl-3.2d",
}
```

#### Specific nuclear data on the Nuclide

You can also avoid setting ```yamc.neutron_cross_section_data``` entirely and specify the nuclear data to use on the Nuclide object directly.

This can be done using a nuclear data library.
```python
nuclide = yamc.Nuclide('Li6')
nuclide.read_nuclear_data('endf-b8.1')
```

Alternatively a specific nuclear data path

```python
nuclide = yamc.Nuclide('Li6')
nuclide.read_nuclear_data('tests/Li6.arrow')
```

Or a directory path (resolves to `tests/Li6.arrow`)

```python
nuclide = yamc.Nuclide('Li6')
nuclide.read_nuclear_data('tests')
```

#### Sample interacting reaction

Once the interacting nuclide is known then the reaction can be sampled using ```Nuclide.sample_reaction()```.

```python
import yamc
nuc = yamc.Nuclide('Li6')
nuc.read_nuclear_data('tests/Li6.arrow')
reaction = nuc.sample_reaction(energy=1.0, temperature='294', seed=42)
if reaction is not None:    # None where no reaction can be sampled
    print(f'interacting reaction is MT {reaction["mt_number"]}')
```

### Transmutation data

For transmutation (``Material.transmute()``), coupled transport-transmutation (``Model.simulate_transmutation()``) and shutdown dose rate (D1S), YAMC also needs a transmutation network: the chain of reactions, decays and decay-photon sources used to evolve nuclide inventories.

The network is assembled from four independently sourced parts, each set globally to a library keyword (its subsection's files are downloaded from the library's current release on first use, verified and cached) or a path to a local chain directory:

<!-- doctest: skip -->
```python
# decay / reactions / fission_yields have no default: set each part the
# calculation reads (keyword or path). Parts may come from different libraries.
yamc.transmutation_decay_data     = "endf-b8.1"
yamc.transmutation_reactions      = "endf-b8.1"
yamc.transmutation_fission_yields = "endf-b8.1"

# Isomeric branching is a separate overlay part; sourced (when available) from a
# TENDL release. Unset means no branching overlay is applied.
yamc.transmutation_branch_ratios  = "tendl-2025"

# The reactions network topology + Q values can also come from a TENDL release,
# which covers far more parent nuclides than endf-b8.1:
yamc.transmutation_reactions      = "tendl-2017"

# A value can also be a path to a local chain directory that holds decay/,
# reactions/ and fission_yields/ subsection subdirectories, or to a whole
# library folder (see "A whole library in one folder" above):
yamc.transmutation_decay_data = "/path/to/transmutation-endf-b8.1-sfr.arrow"
```

A reactions or fission-yields folder must hold its ``reactions.arrow`` or
``fission_yields.arrow``: one without it is refused, naming the folder, rather
than read as a network with no reactions or no yields. Set the part to ``False``
(below) to leave it out.

See [Nuclear data libraries](nuclear_data_libraries.md) for which library provides which subsection. Setting a part to a library that does not publish it fails fast with a message listing that library's available subsections.

A part can also be turned off, which is the correct setting for a decay-only
calculation or for a material in which nothing fissions:

<!-- doctest: skip -->
```python
yamc.set_transmutation_reactions(False)       # decay only, no reaction topology
yamc.set_transmutation_fission_yields(False)  # nothing here fissions
```

``False`` is not the same as leaving a part unset, which is refused by any
calculation that reads it. Read back, each setting is what was assigned: the
source, ``False`` when off, or ``None`` when unset. A rate that then needs the part that was switched off is refused when the
burnup matrix is built, naming the nuclide and the reaction, rather than being
solved as though the reaction produced nothing, or losing its fission products
while still burning the nuclide.

The transmutation APIs and the decay-photon post-processing (``PulseSchedule.time_correct_tally``) read these settings; there is no per-call override. What each calculation reads:

| calculation | reads |
| --- | --- |
| activity, decay heat, decay photon lines and contact dose of a result, and a schedule that only cools | ``transmutation_decay_data`` |
| a D1S run (``use_decay_photons=True``), ``Model.radionuclides()`` and ``PulseSchedule.time_correct_tally`` | ``transmutation_decay_data`` and ``transmutation_reactions`` |
| a schedule with an irradiating step: ``Material.transmute()``, ``yamc.transmute()`` and ``Model.simulate_transmutation()`` | all three |

