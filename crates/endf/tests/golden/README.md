# Golden files

Each `.txt.xz` here is a reference dump of what the **Python** reader produces
for one ENDF evaluation. `tests/golden.rs` reads every one of them, runs the Rust
reader over the evaluation the `SOURCE` line names, and compares.

This is what makes the port safe: the Rust crate is not being written against
the ENDF-102 manual alone, it is being held to a reader that already works.

## ACE fixtures

A `.ace` fixture is dumped differently (ACE has tables, not materials) so its
golden opens with `KIND ace` and the Rust side reads it through
`endf::ace` instead. The XSS array runs to hundreds of thousands of numbers, so
the dump records a spread across it plus both ends and every JXS entry point,
which is where a consumer actually looks. Everything else is recorded in full.

On top of the raw arrays, the ACE dump walks the blocks a transport code
reads: every locator in AND (the angular distributions), the whole of DLW (the
joint angle-energy distributions, following the linked list each reaction
carries), and every reaction the table holds, elastic scattering included.
That turns the ACE fixtures into a check on the interpretation, not only on the
numbers.

## Compression

Fixtures and dumps are both stored xz-compressed. An evaluation is highly
repetitive: the fixtures go 4.1 MB to 655 KB and the dumps 5.3 MB to 748 KB,
about six and seven to one. The Python side reads them through
`endf.fileutils.open_text`, which handles `.xz` and leaves anything else alone;
the Rust side reads them with `lzma-rs`, a pure-Rust **dev-dependency**, so the
`endf` crate stays dependency-free for anything that uses it.

Nothing else changes: `python tools/dump_golden.py` writes `.txt.xz` and the
dumps are still byte-reproducible.

## The chain golden

`chain.txt.xz` is the odd one out: a depletion chain is the join of three
sub-libraries, so its golden names all of them with `DECAY`, `NEUTRON` and
`REACTION` lines instead of a single `SOURCE`. It is written by
`tools/dump_chain_golden.py` rather than by the main dumper, and the Rust
harness recognises it by `KIND chain`.

The ten decay evaluations behind it were chosen to close every path the chain
follows, except Cs137's, whose barium daughters are deliberately absent so
that the stand-in walk of `replace_missing` is exercised.

## Adding an evaluation

1. Compress the file and drop it in `tests/`:
   `python -c "import lzma,sys,pathlib; p=pathlib.Path(sys.argv[1]); pathlib.Path(str(p)+'.xz').write_bytes(lzma.compress(p.read_bytes(), preset=9))" file.endf`
2. `python tools/dump_golden.py` (or pass the one path to regenerate just it).
3. `cargo test -p endf`.

Nothing in the Rust test needs changing, it discovers golden files and follows
`SOURCE`.

## What is compared

| Record | Covers |
|---|---|
| `MATERIALS`, `MAT` | Multi-material files, material numbers |
| `SECTION mf mt n` | Section splitting, for **every** MF including unported ones |
| `production/…`, `reaction/…`, `nuclide/…` | The derived views: the MF=8/9/10 join, every reaction gathered from its files, and the nuclide those reactions belong to |
| `MF3 …`, `BP`, `INT`, `X`, `Y` | Parsed values, compared **exactly** |
| `EVALX` / `EVALY` | Interpolation, compared to 1e-12 relative |

Values are written as the shortest round-tripping decimal and both readers parse
decimals with correct rounding, so parsed values are compared bit-for-bit. Only
computed values use a tolerance, and `is_interpolated` in `golden.rs` lists
exactly which: the interpolation samples, because logs and exps need not round
identically in the two languages; the MF=10 yields, which divide two
interpolated cross sections; the propagated uncertainties, where `uncertainties`
accumulates a variance and takes its square root; and the forward-scattered
fraction with the removal cross section that folds it in, because NumPy
re-associated the Clenshaw recurrence between 2.2 and 2.4 and the two give
different last bits. Nothing that comes off the file is compared loosely.

`python tools/dump_golden.py --check` compares the stored dumps against freshly
generated ones under the same rules, which is what CI runs. It compares the dump
*text* rather than the compressed bytes, two xz encoders can write the same
content differently, and that says nothing about the reader.

The `SECTION` lines matter more than they look: they hold the section splitter
to the Python reader across files that have no Rust parser yet, so a new
evaluation is useful coverage the day it is added, long before every MF is
ported.

## Coverage still wanted

Every ENDF file the Python package parses now has a Rust parser, and all but one
are exercised by a fixture. The exception is

    MF 40 (radionuclide production covariances)

which is structurally complete and unverified end to end, though the MF33
subsection parser it delegates to is covered. The list is pinned in `golden.rs`
as `UNCOVERED_BY_ANY_FIXTURE` and checked, so it cannot drift in either
direction: the test fails both when a fixture starts covering one, and when a
new parser arrives without coverage.

`MF2` is worth a line of its own. It has real Reich-Moore parameters from Fe56
and U235, a Case C unresolved region from U235, and a synthetic multi-level
Breit-Wigner section, but Adler-Adler, R-matrix limited (LRF=7) and unresolved
Cases A and B are still untested. Cases A and B are additionally unreachable
through the current dispatch; see issue #15.

The distribution shapes are tracked the same way, in `DISTRIBUTION_SHAPES`:
every angular, energy and joint angle-energy shape the dumpers can write has to
appear in some golden file, or the test names the one that does not. Where no
real file small enough to keep as a fixture holds a shape, one is built.
`tools/make_urr_ace.py`, `tools/make_laws_ace.py` and
`tools/make_denormal_ace.py` write ACE tables; `tools/make_nfy_endf.py` and
`tools/make_shapes_endf.py` write ENDF evaluations on top of the record writer
in `tools/endf_writer.py`. The values are invented; the layout is the format's,
which is the part both readers are being held to.

Trimming a fixture down to the sections that matter is what `tools/trim_endf.py`
is for. A full evaluation runs to tens of megabytes, most of it covariance
data, U235 is 36 MB whole and 451 KB with ten sections kept.

### Fixtures present

| Fixture | Covers |
|---|---|
| `n-095_Am_244` | MF1 (incl. MT458), MF2 LRF=0, MF3, MF4, MF5 LF=7 |
| `n-095_Am_242_trimmed` | MF1, a metastable target |
| `n-049_In-115_trimmed` | MF3, MF8, MF9, MF10: isomer production |
| `n-077_Ir_191_trimmed` | MF1, MF3 (incl. MT3), MF8, MF9, MF10: TENDL-2017 (n,2n) partials that sum to less than MF3 |
| `n-054_Xe_136_trimmed` | MF1, MF3 |
| `n-003_Li_006_trimmed` | MF6 LAW=2 and LAW=4, MF12, MF14, MF33 |
| `n-026_Fe_056_trimmed` | MF2 Reich-Moore, MF6 LAW=1, MF12/14, MF33 |
| `n-092_U_235_trimmed` | MF2 Reich-Moore + Case C URR, MF5 LF=5, MF8, MF10, MF15, MF34, delayed neutron groups |
| `photoat-001_H_000` | MF23, MF27 |
| `atom-001_H_000` | MF28 |
| `e-001_H_000` | MF23, MF26 in all three laws |
| `tsl-s-CH4` | MF7 MT=2 and MT=4 |
| `dec-049_In_116m1` | MF8 MT=457 decay data: four spectra, beta- only |
| eight more `dec-*` | The decay evaluations that close the chain fixture |
| `dec-049_In_116m2` | MF8 MT=457 decay data: an isomeric transition down to m1 |
| `dec-072_Hf_177m1` | MF8 MT=457 decay data: an isomeric transition whose average energies exceed its Q |
| `Li6.ace` | An ACE Type 1 table; AND in all three shapes, DLW laws 3, 33 and 44, 15 reactions with photon production |
| `synthetic-urr.ace` | The unresolved resonance block, which no small real table has |
| `synthetic-laws.ace` | DLW laws 2, 4, 7, 9, 11, 61 and 66 |
| `synthetic-denormal.ace` | The float form NJOY writes for a denormal, `6.10562372605-318` |
| `synthetic-nfy.endf` | MF8 MT=454 and MT=459, the fission product yields |
| `synthetic-shapes.endf` | MF2 LRF=2 Breit-Wigner, MF5 LF=12 Madland-Nix, MF6 LANG=2 and LAW=6, MF13 |

### Fixtures still wanted

- **MF40**, the one parser nothing exercises. It needs an evaluation with
  radionuclide production covariances, and none small enough to keep here has
  them. Unlike the shapes below, MF40 reuses the MF33 subsection parser that
  Li6, Fe56 and U235 do cover, so what is untested is the wrapper around it.
- **MF2 formalisms beyond Reich-Moore and Breit-Wigner**: R-matrix limited
  (LRF=7), and unresolved Cases A and B. Note that Cases A and B cannot be
  reached at all through the current dispatch (see issue #15) so a fixture
  alone will not cover them.
- **Adler-Adler (LRF=4)**, which both readers reject rather than parse. A
  fixture would only pin that rejection.
- **ACE law 5**, which the Python reader cannot read at all (see issue #19),
  so there is nothing to compare against. Every other ACE law is covered.
- **A second ACE table of the same nuclide at another temperature**, which is
  what `add_temperature_from_ace` exists for. Only the "already present" path
  is exercised.
- **An ACE photoatomic table**, so `IncidentPhoton::from_ace`, the Compton
  profiles and subshell photoelectric cross sections it reads, is written but
  unexercised.
- **A fissile ACE table.** Li6 has no NU block, so the ACE fission path,
  prompt and total nu, the delayed groups and their probabilities, is
  unexercised. So is the URR block on a real table, and MFTYPE=13 photon
  production.
- **A delayed neutron group whose applicability varies with energy.** U235
  gives each group a constant share, which is the usual case; the branch that
  takes the product on the union of two grids is unexercised.
- **Other libraries.** Everything here is ENDF/B-VIII.0 except the ACE table,
  which is TENDL-2023.1. JEFF-4.0, JENDL-5 and TENDL-2025 differ in which
  optional records they write and how strictly they follow the format, which is
  exactly what a format reader gets wrong.

## A note on size

Fixtures are trimmed with `tools/trim_endf.py` rather than added whole; a full
evaluation with covariances runs to tens of megabytes. If one ever has to be
added whole, the dump format should grow a digest record, a hash over a table
rather than its values, so that a large evaluation costs a line instead of a
megabyte. Small fixtures should stay fully enumerated: an exact diff points at
the value that broke, a hash only says something did.
