# Getting started

## Prerequisites

* Python 3.9+
* [endf-python](https://github.com/paulromano/endf-python) for reading ENDF and
  ACE files
* NJOY is required for the ENDF neutron route, which is the default, to
  reconstruct the resonances and Doppler broaden the cross sections. Converting a
  pre-processed ACE table (`source_format="ace"`) needs no NJOY, but yields no
  MT 901 (heating-local) and only the temperature the table was built at.

### Which NJOY

For ENDF/B and TENDL, upstream [NJOY2016](https://github.com/njoy/NJOY2016) is the
right choice.

**FENDL needs [IAEA-NDS/NJOY2016](https://github.com/IAEA-NDS/NJOY2016) instead.**
FENDL is distributed by the IAEA Nuclear Data Section, and that fork exists to
"keep track of modifications of the official NJOY2016 version to ensure the proper
processing of the nuclear data libraries distributed by the Nuclear Data Section of
the IAEA". Its changes are described in
`Updating NJOY2016.72 for IAEA-NDS libraries.docx` in that repository rather than in
its README. Processing FENDL ENDF files with upstream NJOY will produce ACE data
that does not match the official FENDL release.

Two version-sensitive details are worth knowing whichever build you use:

* NJOY2016.75 changed how ACER writes the ZAID for metastable nuclides and added an
  option to choose between `za` regardless of state (the long-standing default) and
  the MCNP rule `za + 300 + s * 100`. The converter assumes the default, and adjusts
  the ZAID itself for metastable targets. Selecting the other convention would make
  that adjustment apply twice.
* NJOY2016.77 fixed the handling of background cross sections in the unresolved
  resonance region for reactions other than total, elastic, fission and capture,
  which matters for LRF=7 evaluations.

## Installation

Not on PyPI yet (see
[yamc#391](https://github.com/fusion-neutronics/yamc/issues/391)), so install
from source.

```bash
pip install "nuclear_data_to_arrow @ git+https://github.com/fusion-neutronics/yamc#subdirectory=packages/nuclear_data_to_arrow"
```

That is enough for the readers, the schemas and the manifest, which parse no
ENDF. **The writers need `endf`, and it must be the fork**: PyPI's `endf` has
none of the features they use, and `endf.incident_photon` is missing from it
outright. `endf` is deliberately not a declared dependency, because a
dependency list cannot name a git branch in a way a published wheel can honour
and naming a bare `endf` named the wrong package
([yamc#525](https://github.com/fusion-neutronics/yamc/issues/525)). Install it
yourself before converting anything:

```bash
pip install "endf @ git+https://github.com/shimwell/endf-python@local-develop"
```

Without it, importing the package still works and every `convert_*` and
`export_*` call raises an `ImportError` naming that command.

Working inside a checkout of the monorepo instead:

```bash
pip install -e packages/nuclear_data_to_arrow[dev]
pip install "endf @ git+https://github.com/shimwell/endf-python@local-develop"
```

## Quick example

Convert a single ENDF neutron file:

```python
from nuclear_data_to_arrow import convert_neutron

convert_neutron(input_path="n-092_U_235.endf", output_dir="output/",
                library="endfb-8.0")
```

This creates `output/U235.arrow/` containing:

```text
output/U235.arrow/
├── version.json
├── nuclide.arrow
├── reactions.arrow       ← includes synthesized MTs 1, 3, 4, 27, 101
├── products.arrow
├── distributions.arrow
├── fast_xs.arrow         ← FastXSGrid lookup tables
├── urr.arrow             (if unresolved resonance data exists)
├── total_nu.arrow        (if fission data exists)
└── fission_photon.arrow  (if fission energy release exists)
```

Each `.arrow` file is an Arrow IPC file that can be memory-mapped for
zero-copy reads.  The `version.json` file records the format version,
source library, and converter version.

## What's in the output?

The `.arrow/` format is **simulation-ready**.  Unlike raw Arrow exports,
it includes all the post-processing that yamc would otherwise do at load time:

Hierarchical MT synthesis
: MTs 1 (total), 3 (non-elastic), 4 (inelastic), 27 (absorption), and
  101 (disappearance) are pre-computed from constituent reactions.

FastXSGrid lookup tables
: A log-binned index (8000 bins) over the energy grid enables O(1) cross-section
  retrieval during particle transport.  The 4-column XS array stores total,
  absorption, scattering, and fission cross sections.

Log-space photon data
: Photon energy grids and cross sections are stored in both linear and
  log space.  Compton profile CDFs are pre-computed via trapezoidal
  integration.

## Running the tests

```bash
pytest packages/nuclear_data_to_arrow/tests
```

The suite is hermetic: it vendors an Li6 ACE table and an Fe photoatomic and
atomic-relaxation ENDF pair, so it needs neither NJOY nor a full library and
runs in a few seconds. To widen coverage on a machine that holds a full
library, set `NUCLEAR_DATA_DIRS` (colon separated) or point
`OPENMC_CROSS_SECTIONS` at a `cross_sections.xml`.
