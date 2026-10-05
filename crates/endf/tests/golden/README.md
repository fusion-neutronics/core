# Golden files

Each `.txt.xz` here is a dump of what the reader returns for one fixture,
every value as a `path -> value` line. `tests/golden.rs` reads every one of
them, runs the reader over the evaluation the `SOURCE` line names, and
compares.

The goldens were first written by the Python endf reader this crate was ported
from, and the port was held to it value for value. When the Python reader was
retired the port matched it on all 70,260 values across 58 goldens, exactly
except for two last-bit differences in computed values (a decay constant and
an MF=10 yield, 1.8e-16 relative). The goldens were then rewritten by this
reader, and are now snapshots it owns: they catch any change in what the
reader returns for a real evaluation, and reviewing a regenerated golden's
diff is where such a change is judged. Correctness of the physics built on top
is checked against NJOY, not here.

Regenerate every golden (unchanged ones are left alone):

    cargo test -p endf --test golden -- --ignored regenerate_goldens

`every_fixture_has_a_golden` fails, naming the file, when a fixture has no
golden or a golden has no fixture.

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
about six and seven to one. The tests read and write them with `lzma-rust2`, a
pure-Rust **dev-dependency**, so the `endf` crate stays dependency-free for
anything that uses it. The comparison is on the decompressed text, never the
compressed bytes, and regenerating leaves a golden whose text is unchanged
untouched, so the bytes do not churn.

## The chain golden

`chain.txt.xz` is the odd one out: a depletion chain is the join of three
sub-libraries, so its golden names all of them with `DECAY`, `NEUTRON` and
`REACTION` lines instead of a single `SOURCE`. The evaluations and reactions
are the `CHAIN_*` lists in `golden.rs`, and the harness recognises the golden
by `KIND chain`.

The ten decay evaluations behind it were chosen to close every path the chain
follows, except Cs137's, whose barium daughters are deliberately absent so
that the stand-in walk of `replace_missing` is exercised.

## Adding an evaluation

1. Compress the file and drop it in `crates/endf/fixtures/`:
   `xz -9 -k file.endf`
2. `cargo test -p endf --test golden -- --ignored regenerate_goldens`
3. Review the new golden, then `cargo test -p endf`.

Nothing in the Rust test needs changing, it discovers fixtures and goldens.

## What is compared

| Record | Covers |
|---|---|
| `MATERIALS`, `MAT` | Multi-material files, material numbers |
| `SECTION mf mt n` | Section splitting, for **every** MF including unported ones |
| `production/…`, `reaction/…`, `nuclide/…` | The derived views: the MF=8/9/10 join, every reaction gathered from its files, and the nuclide those reactions belong to |
| `MF3 …`, `BP`, `INT`, `X`, `Y` | Parsed values, compared **exactly** |
| `EVALX` / `EVALY` | Interpolation, compared to 1e-12 relative |

Values are written as the shortest round-tripping decimal and parsed with
correct rounding, so parsed values are compared bit-for-bit. Only computed
values use a tolerance, and `is_interpolated` and `is_log_law_cdf` in
`golden.rs` list exactly which: those that go through `ln`, `exp` and similar
functions, which are not correctly rounded and differ in the last bit between
the maths libraries of the platforms CI runs on. Nothing that comes off the
file is compared loosely.

The `SECTION` lines matter more than they look: they hold the section splitter
across every file, including any MF that has no parser, so a new evaluation is
useful coverage the day it is added.

## Coverage still wanted

Every ENDF file with a parser is exercised by a fixture. MF 32 was the last:
the Python reader never parsed it, so it could not have a golden until the
goldens became this reader's own. Its fixtures cover LCOMP=0, LCOMP=1 general,
LCOMP=2 compact (Reich-Moore and R-matrix limited) and the unresolved range;
LCOMP=1 for R-matrix limited has none. `tests/mf32_tapes.rs` also walks every
MF=32 section of six libraries.

It is pinned in `golden.rs` as `UNCOVERED_BY_ANY_FIXTURE` and checked, so the
list cannot drift in either direction: the test fails both when a fixture
starts covering one, and when a new parser arrives without coverage.

`MF2` is worth a line of its own. It has real Reich-Moore parameters from Fe56,
U235 and Pb208, R-matrix limited (LRF=7) from Cl35, Cu65, W186 and V51, a Case
C unresolved region from U235, and multi-level Breit-Wigner from Bi209 and a
synthetic section, but Adler-Adler and unresolved Cases A and B are still
untested. Cases A and B are additionally unreachable
through the current dispatch, which tests LRF where the format uses LRU.

The distribution shapes are tracked the same way, in `DISTRIBUTION_SHAPES`:
every angular, energy and joint angle-energy shape the dumpers can write has to
appear in some golden file, or the test names the one that does not. Where no
real file small enough to keep as a fixture holds a shape, one is built.
`tools/make_urr_ace.py`, `tools/make_laws_ace.py` and
`tools/make_denormal_ace.py` write ACE tables; `tools/make_nfy_endf.py` and
`tools/make_shapes_endf.py` write ENDF evaluations on top of the record writer
in `tools/endf_writer.py`. The values are invented; the layout is the format's,
which is the part the reader is being held to.

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
| `n-041_Nb_093_tendl2017_trimmed` | MF1, MF3, MF8, MF9, MF10, MF33, MF40: TENDL-2017 radionuclide production covariance |
| `dec-041_Nb_092`, `dec-041_Nb_092m1`, `dec-041_Nb_093m1` | MF8 MT=457 decay data: the isomer table the Nb93 production levels resolve against |
| `n-054_Xe_136_trimmed` | MF1, MF3 |
| `n-003_Li_006_trimmed` | MF6 LAW=2 and LAW=4, MF12, MF14, MF33 |
| `n-026_Fe_056_trimmed` | MF2 Reich-Moore, MF6 LAW=1, MF12/14, MF33 |
| `n_2825_28-Ni-58_trimmed.fendl32d` | MF33 LB=0, 1, 4 and 5 from FENDL-3.2d: LB=1 tables with odd and even NP, and the only LB=4 block on any tape |
| `n-024_Cr_052_trimmed.endfb81` | MF33 LB=0, 1 and 8 from ENDF/B-VIII.1: the (n,p) tables whose upper half a split at NT - NP rather than 2*(NP - LT) drops |
| `n-092_U_235_trimmed` | MF2 Reich-Moore + Case C URR, MF5 LF=5, MF8, MF10, MF15, MF34, delayed neutron groups |
| `photoat-001_H_000` | MF23, MF27 |
| `atom-001_H_000` | MF28 |
| `e-001_H_000` | MF23, MF26 in all three laws |
| `tsl-s-CH4` | MF7 MT=2 and MT=4 |
| `dec-049_In_116m1` | MF8 MT=457 decay data: four spectra, beta- only |
| eight more `dec-*` | The decay evaluations that close the chain fixture |
| `dec-049_In_116m2` | MF8 MT=457 decay data: an isomeric transition down to m1 |
| `dec-072_Hf_177m1` | MF8 MT=457 decay data: an isomeric transition whose average energies exceed its Q |
| `dec-098_Cf_252.jeff40` | MF8 MT=457 decay data from JEFF-4.0: a linear-linear photon continuum (LCON=2) beside the lines, and a log-linear neutron continuum |
| `dec-089_Ac_227.jeff40` | MF8 MT=457 decay data from JEFF-4.0: gamma and x-ray lines at a shared energy, and energies repeated inside one spectrum |
| `dec-092_U_235` | MF8 MT=457 decay data: a fissioning parent, so the `synthetic-nfy.endf` yields reach a chain |
| `Li6.ace` | An ACE Type 1 table; AND in all three shapes, DLW laws 3, 33 and 44, 15 reactions with photon production |
| `synthetic-urr.ace` | The unresolved resonance block, which no small real table has |
| `synthetic-laws.ace` | DLW laws 2, 4, 7, 9, 11, 61 and 66 |
| `synthetic-denormal.ace` | The float form NJOY writes for a denormal, `6.10562372605-318` |
| `synthetic-nfy.endf` | MF8 MT=454 and MT=459, the fission product yields |
| `synthetic-shapes.endf` | MF2 LRF=2 Breit-Wigner, MF5 LF=12 Madland-Nix, MF6 LANG=2 and LAW=6, MF13 |
| `n-094_Pu_244_mf2_mf32` | MF32 LCOMP=0 (compatible) |
| `n-066_Dy_158_mf2_mf32` | MF32 LCOMP=1 general covariance blocks |
| `n-011_Na_023_mf2_mf32`, `n-090_Th_232_mf2_mf32` | MF32 LCOMP=2 compact Reich-Moore; Th232 also an unresolved range |
| `n-017_Cl_035_mf2_mf32`, `n-029_Cu_065_mf2_mf32`, `n-074_W_186_mf2_mf32` | MF2 LRF=7 and MF32 LCOMP=2 compact R-matrix limited |
| `n-045_Rh_103_mf2_mf32` | MF32 compact R-matrix limited plus an unresolved range |
| `n-082_Pb_208_mf2`, `n-092_U_235_mf2`, `n-023_V_051_mf2`, `n-083_Bi_209_mf2` | MF2 alone: Reich-Moore, LRF=7 with APE and APT both given, multi-level Breit-Wigner |

### Fixtures still wanted

- **Unresolved Cases A and B.** They cannot be reached at all through the
  current dispatch (it tests LRF where the format uses LRU) so a fixture alone
  will not cover them.
- **MF32 LCOMP=1 for R-matrix limited**, the one MF32 layout no fixture has.
- **Adler-Adler (LRF=4)**, which the reader rejects rather than parses. A
  fixture would only pin that rejection.
- **ACE law 5**, which the reader refuses by name. Every other ACE law is
  covered.
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
- **Other libraries.** Nearly everything here is ENDF/B-VIII.0. The exceptions
  are the ACE table (TENDL-2023.1), Ir191 (TENDL-2017), the Sn111 decay file
  (JENDL-5) and the two MF33 fixtures (FENDL-3.2d Ni58 and ENDF/B-VIII.1
  Cr52). JEFF-4.0, JENDL-5 and TENDL-2025 differ in which
  optional records they write and how strictly they follow the format, which is
  exactly what a format reader gets wrong.

## A note on size

Fixtures are trimmed with `tools/trim_endf.py` rather than added whole; a full
evaluation with covariances runs to tens of megabytes. If one ever has to be
added whole, the dump format should grow a digest record, a hash over a table
rather than its values, so that a large evaluation costs a line instead of a
megabyte. Small fixtures should stay fully enumerated: an exact diff points at
the value that broke, a hash only says something did.
