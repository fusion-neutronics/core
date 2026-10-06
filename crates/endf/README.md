# endf

A reader for ENDF-6 evaluated nuclear data files and ACE Type 1 tables.

No dependencies, no `unsafe`, no Python. It describes the formats and nothing
more: a simulation-ready projection of the data (reconstructed resonances,
summed reactions, unionised grids, an Arrow schema) belongs in a consumer
built on top, which is then free to depend on whatever it needs without that
cost reaching everyone who just wants to read a file.

```rust
use endf::{IncidentNeutron, Interpretation, Material};

let material = Material::from_file("n-095_Am_244.endf")?;

// The whole file, section by section, keyed by (MF, MT).
let capture = material.mf3(102).expect("a capture cross section");
println!("{} barns at 0.0253 eV", capture.sigma.eval(0.0253));

// Or the high-level view the sublibrary calls for.
if let Interpretation::IncidentNeutron(nuclide) = material.interpret()? {
    println!("{} has {} reactions", nuclide.name(), nuclide.reactions.len());
}
# Ok::<(), endf::Error>(())
```

ACE tables are read the same way, and produce the same types:

```rust
use endf::{get_tables, IncidentNeutron, MetastableScheme};

let tables = get_tables("Li6.ace")?;
let nuclide = IncidentNeutron::from_ace(&tables[0], MetastableScheme::default())?;
# Ok::<(), endf::Error>(())
```

## What it reads

| | |
|---|---|
| ENDF files | MF 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 23, 26, 27, 28, 32, 33, 34, 40 |
| ACE | Type 1 tables: the ESZ, AND, DLW, MTR, LSIG/SIG, TYR and URR blocks |
| Derived | reactions, products and their distributions, incident neutron and photon data, decay data, fission product yields, radionuclide production, depletion chains |

Each file is read in every representation it defines, not only the ones common
evaluations happen to use.

## Correctness

This is a port of [`endf-python`](https://github.com/paulromano/endf-python),
and it was held to it rather than to the ENDF-102 manual alone: the Python
reader dumped every value it produced for each fixture, the Rust reader built
the same `path -> value` map, and the two were compared whole. When the Python
reader was retired they agreed on all 70,260 values across 58 evaluations, ACE
tables and a depletion chain, exactly except for two last-bit differences in
computed values. The goldens are now written by this reader and kept as
snapshots, MF=32 included, so any change in what it returns for a real
evaluation shows up as a diff to review.

See `tests/golden/README.md` for what is covered and what is not.

MF=32 is also held to the tapes: `tests/mf32_tapes.rs` reads every MF=32
section of ENDF/B-VIII.1, JEFF-4.0, JENDL-5.0, TENDL-2017, TENDL-2025 and
FENDL-3.2d to its SEND record and checks the counts of what it read against an
independent survey of the same files. It needs the libraries locally, so it is
`#[ignore]`d by default.

## Minimum supported Rust version

1.74, checked in CI. Raising it is a breaking change.
