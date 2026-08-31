# nuclear_data_to_arrow

Converts evaluated nuclear data (ENDF-6 or ACE) into the simulation-ready
[Apache Arrow IPC](https://arrow.apache.org/docs/format/Columnar.html#ipc-file-format)
directories that `yamc` and `yani` read.

The output is columnar, fixed-width and zero-copy mappable, with the work that
would otherwise happen at every simulation start (hierarchical MT synthesis,
FastXSGrid lookup tables, log-space photon cross sections, Compton CDFs) done
once at conversion time.

Two families of output, written independently:

- **Transport data**, `{nuclide}.arrow/` and `{element}.arrow/`: cross sections,
  secondary distributions, URR tables, fission nu.
- **Transmutation networks**, `transmutation_{library}.arrow/`: four separately
  library-sourced subsections (`decay`, `reactions`, `fission_yields`,
  `branching`), so a consumer can assemble a chain from a different library per
  subsection.

The transmutation half has no dependency on the transport format, which is what
lets a transport-free activation package consume it on its own.

## Install

```bash
pip install -e packages/nuclear_data_to_arrow
```

Note that the `endf` dependency currently needs
[shimwell/endf-python](https://github.com/shimwell/endf-python) rather than the
`endf` on PyPI: `from_njoy`, `Chain`, `radionuclide_production`, `isomer_table`
/ `level_to_isomeric_state`, `univariate` and the Compton profile helpers are all
fork-only. The ACE reader is not: `endf` 0.1.12 on PyPI ships `ace.py` and
`IncidentNeutron.from_ace`. The module that breaks a PyPI install is
`endf.incident_photon`, whose absence fails at import rather than at first use.

NJOY is needed only for `source_format="endf"`, which reconstructs resonances
and Doppler broadens to each requested temperature, and is the only route that
yields MT 901 (heating-local). `source_format="ace"` reads a pre-processed table
and needs no NJOY, at the cost of a fixed temperature and no MT 901.

### Choosing which NJOY

Pass `njoy_exec=` to pin the executable
per library. This is not a preference. FENDL is processed by the
[IAEA-NDS fork](https://github.com/IAEA-NDS/NJOY2016) with local modifications,
and upstream NJOY does not reproduce the official release: measured on FENDL
3.2d against LANL 2016.78, the fork changes URR probability tables by up to 100%
(La139) and MT 301 heating / MT 444 damage by up to 77% below roughly 1 keV.
Both builds succeed, so the wrong choice is quiet.

```python
convert_neutron("n-026_Fe_056.endf", "out/")                        # njoy from PATH
convert_neutron("f-026_Fe_056.endf", "out/", njoy_exec="/opt/njoy-iaea/njoy")
```

## Python API

```python
from nuclear_data_to_arrow import convert_neutron, convert_photon

convert_neutron("n-092_U_235.endf", "output/", library="endfb-8.1")
convert_neutron("92235.710nc", "output/", source_format="ace", library="endfb-8.1")
convert_photon("photoat-026_Fe_000.endf", "output/",
               atom_path="atom-026_Fe_000.endf", library="endfb-8.1")
```

### Stamping a release with `data_version`

Every converter takes a `data_version`, written into the `version.json` (or
`provenance.json`) of what it produces. It identifies the published *release* of
the data, and it is what yamc compares a cached copy against to decide whether a
re-published library has invalidated the cache (yamc issue #366).

```python
convert_neutron("n-092_U_235.endf", "output/", library="fendl-3.2d",
                data_version="2026-08-09.1")
```

It is not `converter_version`. That identifies the code, so two rebuilds from
the same converter carry the same one and are still different data, which is
precisely the case that has to invalidate a cache. The hosted objects are
overwritten in place on a re-publish, so the URL and the cache key are identical
before and after and this stamp is the only thing that differs.

Pass the same value for every file in one release, and a new one for every
rebuild. Publishing a rebuild then takes two steps that belong in the same
release:

1. convert and upload with the new `data_version`
2. add or bump the library's entry in `EXPECTED_DATA_VERSION` in
   `crates/yamc-nuclide/src/storage/url_cache.rs`

Doing only the first leaves existing installs on the old data. Doing only the
second makes every load evict its cache, refetch, and then fail with a message
saying so, which is the intended way round: a publishing mistake is loud rather
than silent.

Left unset it is written as an empty string, which yamc treats as unstamped, and
a library with no `EXPECTED_DATA_VERSION` entry is not checked at all.

## No command line

This package is a library. It has no console scripts and no `python -m` entry
point, because walking a release (which files a library ships, downloading them,
filtering, resuming, sizing a worker pool) is a build concern rather than a
format concern.

Every entry point lives in the
[nuclear_data_generation_scripts](https://github.com/fusion-neutronics/nuclear_data_generation_scripts)
repo and calls this package's functions:

```bash
python -m converters.endf --release viii.1 --njoy-exec "$NJOY_BIN"
python -m converters.transmutation --decay-dir decay/ --fpy-dir nfy/ ...
python -m converters.branching --neutron-dir neutrons/ ...
```

To convert a single file, call the API directly:

```python
from nuclear_data_to_arrow import convert_neutron
convert_neutron("n-092_U_235.endf", "output/", library="endfb-8.1")
```

## Tests

```bash
pytest packages/nuclear_data_to_arrow/tests
```

The suite is hermetic: it vendors an Li6 ACE table and an Fe photoatomic and
atomic-relaxation ENDF pair, and needs neither NJOY nor a full library. Set
`NUCLEAR_DATA_DIRS` (colon separated) or `OPENMC_CROSS_SECTIONS` to widen
coverage on a machine that holds one.

## Building published data

Orchestration of full-library builds (NJOY provisioning, per-library recipes,
Cloudflare R2 upload) lives in the separate `nuclear_data_generation_scripts`
repo, which drives this package.
