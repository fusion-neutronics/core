# Usage

## Python API

The two main functions accept nuclear data source files and write
simulation-ready Arrow output:

```python
from nuclear_data_to_arrow import convert_neutron, convert_photon
```

### Neutron data

From an ENDF file, which is the default and NJOY is invoked automatically:

```python
convert_neutron(
    input_path="n-092_U_235.endf",
    output_dir="output/",
    temperatures=[293.6, 600.0, 900.0],
)
# creates output/U235.arrow/
```

From a pre-processed ACE table, when NJOY is unavailable:

```python
convert_neutron(
    input_path="92235.710nc",
    output_dir="output/",
    source_format="ace",
)
```

Prefer ENDF. It is the only route that produces MT 901 (heating-local), which is
built from two HEATR passes plus the MF=1/458 fission energy release and so cannot
be recovered from an ACE table, and it Doppler broadens to whichever temperatures
you ask for rather than the single one the table was built at. Passing
`temperatures` with `source_format="ace"` raises `ValueError` for that reason.

With a library name (stored in version.json):

```python
convert_neutron(input_path="n-092_U_235.endf", output_dir="output/",
                library="endfb-8.0")
```

### Photon data

Photon files are always ENDF.  If a separate atomic relaxation file is
available, pass it with `atom_path`:

```python
convert_photon(
    input_path="photoat-026_Fe_000.endf",
    output_dir="output/",
    atom_path="atom-026_Fe_000.endf",
    library="endfb-8.0",
)
# creates output/Fe.arrow/
```

For ENDF files containing multiple elements (e.g. FENDL bundles):

```python
from nuclear_data_to_arrow import convert_photon_endf

convert_photon_endf(
    input_path="fendl-photoat.endf",
    output_dir="output/",
    library="fendl-3.2c",
)
# creates output/H.arrow/, output/He.arrow/, ...
```

### Transmutation network

`convert_transmutation` exports a full transmutation network -- decay data,
decay-product sources, transmutation reactions, and fission product yields -- to
a single `transmutation_{library}.arrow/` directory.  Give it the trio of ENDF
input lists used to build the network.

From ENDF source directories (what the release build script does):

```python
from pathlib import Path
from nuclear_data_to_arrow import convert_transmutation

endf = Path.home() / "nuclear_data" / "endfb-viii.1-endf"

convert_transmutation(
    "transmutation_endf_b8.1_sfr.arrow",
    decay_files=list((endf / "decay-version.VIII.1").rglob("*.endf")),
    fpy_files=list((endf / "nfy-version.VIII.1").rglob("*.endf")),
    neutron_files=list((endf / "neutrons-version.VIII.1").rglob("*.endf")),
    branch_ratios="branching_ratios_sfr.json",   # openmc_data format, optional
    library="endfb-8.1",
)
```

The optional `branch_ratios` JSON lets you override reaction-product branching
ratios after the network is built -- useful for sodium-fast-reactor (SFR) vs.
thermal branching conventions.  See `format.md` for the full directory layout
and schemas.

## No command line

This package is a library. It registers no console scripts and has no
`python -m` entry point.

Walking a release (working out which files a library ships, downloading them,
filtering, resuming a partial run, sizing a worker pool) is a build concern
rather than a format concern, so every entry point lives in the
`nuclear_data_generation_scripts` repo instead, as
`converters/{endf,fendl,jeff,tendl,transmutation,branching}.py`. Those drivers
import this package and call it one file at a time:

```bash
python -m converters.endf --release viii.1 --njoy-exec "$NJOY_BIN"
python -m converters.transmutation --decay-dir decay/ --fpy-dir nfy/ ...
python -m converters.branching --neutron-dir neutrons/ ...
```

That repo is also where the release URLs, per-library source filename
conventions and the NJOY flavour each library needs are recorded, so a library
is described in exactly one place.

To convert a single file, use the Python API above.

## Reading Arrow files back

### Through yamc's own loader

This is the check that matters for a conversion, because yamc is the consumer:
a directory can satisfy every schema and still be refused by the loader that
has to read it. `yamc` (and `yani`, which ships the same binding) expose the
Rust readers directly, so no second implementation of the format is involved.

```python
import yamc

summary = yamc.read_nuclide_from_arrow("output/U235.arrow")
assert summary["scope_loaded"] == "full"   # not narrowed to cross-sections only
print(summary["name"], summary["mts"])

element = yamc.read_element_from_arrow("output/Fe.arrow")
assert element["has_compton_profiles"] and element["has_bremsstrahlung"]
```

`scope_loaded` is the one to assert on. The loader does not fail on a directory
holding no transport sections: it narrows a `"full"` request to
cross-sections-only, so a check that only asks whether the call returned has
said nothing about distributions, products or `fast_xs`.

### Through this package's own readers

For inspection of the raw tables, the package reads them back into Python
dicts. These share the writers' vocabulary, so they are the weaker gate:

```python
from nuclear_data_to_arrow import read_neutron_from_arrow, read_photon_from_arrow

data = read_neutron_from_arrow(path="output/U235.arrow")
print(data["version"])                   # format and converter metadata
print(data["nuclide"]["name"])           # "U235"
print(len(data["reactions"]))            # number of reactions (incl. synthesized)
print(data["reactions"][0]["mt"])         # first reaction MT number

# FastXSGrid data
for fxs in data["fast_xs"]:
    print(fxs["temperature"], len(fxs["log_grid_index"]))  # 8001 entries

photon = read_photon_from_arrow(path="output/Fe.arrow")
print(photon["element"]["Z"])            # 26
print(len(photon["element"]["ln_energy"]))  # log-space energy grid
```

## Verification

To confirm that the Arrow output matches the source data within tolerance,
use the verification functions:

```python
import endf
from nuclear_data_to_arrow import export_neutron_to_arrow, verify_neutron

data = endf.IncidentNeutron.from_ace("92235.710nc")
export_neutron_to_arrow(data=data, path="U235.arrow")
assert verify_neutron(data=data, arrow_path="U235.arrow")
```

Verification uses `np.allclose` with `rtol=1e-12` for floating-point
comparisons and additionally checks:

- Synthesized MTs (1, 3, 4, 27, 101) are present and marked redundant
- FastXSGrid: `total == absorption + scattering + fission`
- `log_grid_index` is monotonically non-decreasing with 8001 entries
- `version.json` is present with valid `format_version`
