# Reference outputs

Files here are not golden dumps. A golden dump is a `path -> value` map that
`tests/golden.rs` compares whole; these are single artefacts that a unit test
checks against directly, and they live apart so the golden harness does not
try to parse them.

| File | What it is |
|---|---|
| `njoy-deck.txt.xz` | The NJOY input deck the **Python** `make_ace` composes for `n-095_Am_244` at 293.6 K and 900 K, captured with its `run` stubbed out. `njoy::tests::composes_the_same_deck_as_the_python_package` holds the Rust deck to it byte for byte. |
| `doppler-w186.txt.xz` | W186 cross sections from NJOY's RECONR (0 K) and BROADR (293.6 K, 900 K). `tests/doppler.rs` holds `endf::doppler::broaden` to BROADR with them; see below. |

Regenerate the deck with:

```python
import endf, lzma, unittest.mock as mock
from endf import njoy

material = endf.Material('tests/n-095_Am_244.endf.xz')
captured = {}
def capture(commands, tapein, tapeout, **kwargs):
    captured['commands'] = commands
    raise SystemExit(0)
with mock.patch.object(njoy, 'run', capture):
    try:
        njoy.make_ace('tests/n-095_Am_244.endf.xz', temperatures=[293.6, 900.0],
                      material=material, output_dir='.')
    except SystemExit:
        pass
open('crates/endf/tests/reference/njoy-deck.txt.xz', 'wb').write(
    lzma.compress(captured['commands'].encode(), preset=9))
```

## `doppler-w186.txt.xz`

NJOY2016 (2016.80, commit `ab27c64`) run on ENDF/B-VIII.1 W186
(`n-074_W_186.endf`, MAT 7443): RECONR to a 0 K PENDF, then BROADR to 293.6 K
and 900 K, both at a fractional tolerance of 0.001. With the evaluation as
`tape20`, the deck is

```text
reconr
20 21
'W186 0 K pendf'/
7443 0/
0.001/
0/
broadr
20 21 22
7443 2 0 0 0./
0.001/
293.6 900.0/
0/
stop
```

run as `njoy < input`, which writes the 0 K PENDF to `tape21` and the
broadened one to `tape22`. BROADR broadens to the top of the resolved range,
10 keV. The reference keeps MT=2 and MT=102: the 0 K tables to 1.05 keV,
every point, and the broadened tables to 1 keV, every third point, which keeps
the file near 100 KB and still samples every resonance BROADR resolved. Write
it from the tapes with

```sh
DOPPLER_NJOY_DIR=/path/to/njoy/run \
  cargo test -p endf --test doppler -- --ignored regenerate_njoy_reference
```

`tests/doppler.rs` broadens the 0 K tables at the broadened tables' energies
and compares.

The file was first written with 2016.79 (commit `ac5adf5`); 2016.80 writes it
byte for byte the same.
