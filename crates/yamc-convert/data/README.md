# Auxiliary photon data

None of this is ENDF. The photoatomic sublibrary carries no Compton profiles,
no bremsstrahlung cross sections and no density effect correction, so a
transport code that needs them takes them from separate tabulations. Each file
here is one tabulation, named for what it holds and for the measurement it came
from rather than for the container it used to arrive in.

| file | contents | primary source |
| --- | --- | --- |
| `compton_profiles_biggs1975.txt` | Compton profiles J(pz) per subshell, subshell occupancies, binding energies (eV) | Biggs, Mendelsohn and Mann, *At. Data Nucl. Data Tables* **16** (1975) 201 |
| `density_effect_sternheimer1982.txt` | Mean excitation energy I, subshell occupancies, ionization energies (eV) | NIST ESTAR mean excitation energies; Sternheimer, Seltzer and Berger (1982) |
| `bremsstrahlung_seltzer_berger1986.txt` | Scaled bremsstrahlung differential cross sections, 57 electron energies by 30 reduced photon energies, per element | Seltzer and Berger, *At. Data Nucl. Data Tables* **35** (1986) 345 |

Z = 1 to 100 throughout.

## How they got here

Each file opens with a header naming its own source, so it does not need this
README to explain itself.

### Compton profiles: fetched, first-hand

`tools/fetch_photon_data.py` downloads the Geant4 G4EMLOW data set, verifies it
against a pinned SHA-256, and reads `doppler/p-biggs.dat`,
`doppler/profile-<Z>.dat` and `doppler/shell-doppler.dat`. That archive is the
distribution of the Biggs tables, so the provenance is first-hand and anyone can
rerun it:

```sh
python tools/fetch_photon_data.py            # download, verify, write
python tools/fetch_photon_data.py --check    # compare, write nothing
```

G4EMLOW **6.48** is pinned rather than the newest release. The `doppler`
directory is byte-identical from 6.48 through 8.7, compared file by file, and
6.48 and 9.0 produce identical output here, so the newer archives carry the same
Compton data at 350 MB instead of 25 MB. The pin is recorded with its reason so
it can be revisited if that stops being true. Newer is not the same as different.

A second, independent route exists as a check. `tools/convert_photon_data.py`
rebuilds the same values from the HDF5 that
[shimwell/endf-python](https://github.com/shimwell/endf-python) ships for its
Python reader, and compares rather than writing a rival copy:

```sh
python tools/convert_photon_data.py --from <endf-python>/src/endf/datafiles
```

The two agree line for line. That agreement is what establishes the HDF5 held
that Geant4 dataset and nothing else.

### Density effect: vendored, no primary path

Still built from `density_effect.h5` by `tools/convert_photon_data.py`. NIST
ESTAR publishes mean excitation energies through a web form rather than as a
download, and G4EMLOW's `estar/` directory holds stopping powers with the
density-effect parameter delta against energy, which is a different quantity
from what this file carries. If a machine-readable ICRU-37 or Sternheimer
tabulation turns up, this is the one function that would change.

### Bremsstrahlung: Seltzer's own tabulation, vendored

The file is S. M. Seltzer's BREMSPEC-2 tabulation, National Bureau of Standards,
5 September 1984, the machine-readable form of the tables published in Seltzer
and Berger, *At. Data Nucl. Data Tables* **35** (1986) 345. Its header says so.
It was already whitespace-separated text and is carried unchanged. As a work of
the US federal government it carries no US copyright. Cite Seltzer and Berger
(1986) for it.

The Geant4 G4EMLOW data set also ships Seltzer-Berger data, under `brem_SB/`,
and it is not used here, for two reasons. First, its terms say the data are not
for commercial use and must be used within Geant4. Second, it is not a
different calculation: the tables are identical, digit for digit, to the
PENELOPE 2008 set that NIST supplied, which agrees with this file to its
printed precision everywhere except the last three reduced photon energies at
the tip (k/T of 0.99995 and above) and adds two points at k/T = 0.025 and 0.075.

A newer calculation exists: Poškus's partial-wave BremsLib (CC BY 4.0), which
computes directly the 2 to 30 MeV band that Seltzer and Berger interpolate.
Adopting it would be a physics change and has not been done.

## Why plain text

HDF5 would put a C library in the dependencies of a crate that deliberately has
none, and these are read at runtime, so compressing them would put a
decompressor there instead. The conversion from HDF5 was checked value by value
against the originals: 49,830 Compton profile, binding energy and density effect
values, bit-identical.

## Reading them

`endf::PhotonData::from_files` takes one path per file and
`IncidentPhoton::add_photon_data` attaches the result by atomic number. They are
not embedded in the crate: together they are about 2.4 MB, which does not belong
in every binary that links it, and a consumer reading nuclear data is opening
files anyway.

The bremsstrahlung cross sections are tabulated on 57 electron energies and
resampled onto 200 with a not-a-knot cubic spline, which is what SciPy's
`CubicSpline` does by default. Getting that wrong bends every element's cross
sections near the ends of the grid, so `spline.rs` transcribes SciPy's
arithmetic and the two were compared over 600,000 resampled values: worst
relative difference 3.8e-15.
